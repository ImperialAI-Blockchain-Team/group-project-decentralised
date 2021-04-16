import web3 from './web3';
import { jobs} from "../abi/abi";

const address = '0xE143956c38D133cd1812399cB95AeaF7755f220f';

export default new web3.eth.Contract(jobs, address);