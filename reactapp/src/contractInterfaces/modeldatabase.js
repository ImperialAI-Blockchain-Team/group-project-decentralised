import web3 from './web3';
import { modelDatabase} from "../abi/abi";

const address = '0x200426e147Eae69Ea86bc32ada390e6E3Fd22956';

export default new web3.eth.Contract(modelDatabase, address);
