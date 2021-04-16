import web3 from './web3';
import { modelDatabase} from "../abi/abi";

const address = '0x2064277186250AD5c8EfBFAb0483AF24785F9e59';

export default new web3.eth.Contract(modelDatabase, address);